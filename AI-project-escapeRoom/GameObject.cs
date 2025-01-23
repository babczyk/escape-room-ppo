using System;
using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Graphics;

public class GameObject
{
    public Vector2 Position { get; set; }
    public Vector2 Velocity { get; set; }
    public Vector2 Size { get; set; }
    public bool IsGrounded { get; set; }

    public float gravity { private get; set; }
    private float terminalVelocity;
    private bool wascolliding;
    private Texture2D texture;

    public GameObject(Vector2 position, Vector2 size, float gravity = 9.8f, float terminalVelocity = 100000f)
    {
        Position = position;
        Velocity = Vector2.Zero;
        Size = size;
        this.gravity = gravity;
        this.terminalVelocity = terminalVelocity;
    }

    public void LoadTexture(GraphicsDevice graphicsDevice, Color color)
    {
        // Create a simple colored texture for the object
        texture = new Texture2D(graphicsDevice, (int)Size.X, (int)Size.Y);
        Color[] data = new Color[(int)Size.X * (int)Size.Y];
        for (int i = 0; i < data.Length; ++i) data[i] = color;
        texture.SetData(data);
    }

    public void Update(GameTime gameTime)
    {
        float deltaTime = (float)gameTime.ElapsedGameTime.TotalSeconds;

        // Apply gravity if not grounded
        if (!IsGrounded)
        {
            Velocity += new Vector2(0, gravity);

            // Clamp velocity to terminal velocity
            if (Velocity.Y > terminalVelocity)
                Velocity = new Vector2(Velocity.X, terminalVelocity);
        }

        // Update position
        Position += Velocity * deltaTime;

    }

    public void Draw(SpriteBatch spriteBatch)
    {
        if (texture != null)
        {
            spriteBatch.Draw(texture, Position, Color.White);
        }
    }

    public void ApplyForce(Vector2 force)
    {
        Velocity += force;
    }
    public void ApplyImpulse(Vector2 impulse)
    {
        Velocity = impulse;
    }
    public void Move(Vector2 move)
    {
        Position += move;
    }
    public bool Intersects(GameObject other)
    {
        int bottomOverlap = 0; // Extra overlap on the bottom to prevent sticking to the ground
        // Create a rectangle for this object
        Rectangle thisRect = new Rectangle((int)Position.X, (int)Position.Y, (int)Size.X, (int)Size.Y + (int)bottomOverlap);

        // Create a rectangle for the other object
        Rectangle otherRect = new Rectangle((int)other.Position.X, (int)other.Position.Y, (int)other.Size.X, (int)other.Size.Y);

        // Check for intersection manually
        bool intersects = thisRect.Left <= otherRect.Right &&
                  thisRect.Right >= otherRect.Left &&
                  thisRect.Top <= otherRect.Bottom &&
                  thisRect.Bottom >= otherRect.Top;

        if (intersects)
        {
            wascolliding = true;
        }

        // Check for intersection
        return intersects;
    }

    public bool ExitIntersect(GameObject other)
    {
        // Create a rectangle for this object
        Rectangle thisRect = new Rectangle((int)Position.X, (int)Position.Y, (int)Size.X, (int)Size.Y);

        // Create a rectangle for the other object
        Rectangle otherRect = new Rectangle((int)other.Position.X, (int)other.Position.Y, (int)other.Size.X, (int)other.Size.Y);

        // Check if the rectangles intersect
        if (wascolliding && !thisRect.Intersects(otherRect))
        {
            wascolliding = false;
            return true;
        }

        return false;
    }


    public void StopByWall(Wall wall)
    {

        if (Intersects(wall))
        {
            // Calculate the overlap distances in both X and Y directions
            float overlapX = Math.Min(Position.X + Size.X - wall.Position.X, wall.Position.X + wall.Size.X - Position.X);
            float overlapY = Math.Min(Position.Y + Size.Y - wall.Position.Y, wall.Position.Y + wall.Size.Y - Position.Y);

            // Resolve collision on the axis with the smallest overlap
            if (overlapX < overlapY)
            {
                // Resolve X-axis collision
                if (wall.Position.X < Position.X)
                    Position = new Vector2(wall.Position.X + wall.Size.X, Position.Y);
                else
                    Position = new Vector2(wall.Position.X - Size.X, Position.Y);

                // Stop horizontal movement
                Velocity = new Vector2(0, Velocity.Y);
            }
            else
            {
                System.Console.WriteLine(Velocity + " " + Position + " " + IsGrounded);
                // Resolve Y-axis collision
                if (wall.Position.Y < Position.Y)
                    Position = new Vector2(Position.X, wall.Position.Y + wall.Size.Y);
                else
                {
                    if (Velocity.Y > 0)
                        Velocity = new Vector2(Velocity.X, 0);
                    Position = new Vector2(Position.X, wall.Position.Y - Size.Y);

                    IsGrounded = true; // Stop vertical movement and mark as grounded if appropriate
                }


            }
            System.Console.WriteLine("yes in bound");
        }
        else
            System.Console.WriteLine("exit in bound");
    }
}
