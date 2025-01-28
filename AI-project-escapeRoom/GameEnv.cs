using System.Runtime.Serialization.Formatters;
using AI_project_escapeRoom; // Replace 'YourNamespace' with the actual namespace where Game1 is defined
using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Graphics;
using Microsoft.Xna.Framework.Input;
using System.Linq;
class GameEnvironment
{
    private Game1 game;
    private int maxSteps = 500;
    private int currentStep;

    public GameEnvironment(Game1 game)
    {
        this.game = game;
        this.currentStep = 0;
    }

    public double[] GetState()
    {
        // Example state representation
        return new double[]
        {
            game.player.Position.X / game.widthLevel,
            game.player.Position.Y / game.groundLevel,
            game.box.Position.X / game.widthLevel,
            game.box.Position.Y / game.groundLevel,
            game.IsPressed ? 1.0 : 0.0,
            game.IsOpen ? 1.0 : 0.0
        };
    }

    public (double[], int, bool) Step(int action)
    {
        currentStep++;
        int reward = 0;
        bool IsDone = false;
        // Apply action
        switch (action)
        {
            case 0: game.player.Move(new Vector2(-10, 0)); break; // Move left
            case 1: game.player.Move(new Vector2(10, 0)); break;  // Move right
            case 2: if (game.player.IsGrounded) game.player.ApplyForce(new Vector2(0, -250)); break; // Jump
            case 3: game.player.Grab(game.box); break; // Interact (example)
            case 4: game.player.DropHeldBox(); break; // Interact (example)
        }

        // Improved reward system
        if (game.box.Intersects(game.button))
        {
            reward += 100; // Significant reward for placing the box on the button
            game.IsPressed = true; // Mark button as activated
        }

        // Encourage progress toward the goal
        if (game.IsMovingToward(game.box, game.lastPlayerPosition))
            reward += 2; // Small reward for moving toward the box

        if (game.IsMovingToward(game.button, game.lastPlayerPosition))
            reward += 5; // Reward for moving the box closer to the button

        if (game.IsMovingToward(game.door, game.lastPlayerPosition) && game.IsPressed)
            reward += 10; // Reward for heading toward the door after activating the button

        // Extra rewards for reaching key milestones
        if (game.player.Intersects(game.box))
            reward += 20; // Reward for successfully reaching and interacting with the box

        if (game.player.Intersects(game.door) && game.IsPressed)
        {
            reward += 200; // High reward for completing the goal
            IsDone = true; // Mark the episode as complete
        }

        // Punishments to deter bad behavior
        if (!game.box.Intersects(game.button) && game.player.Intersects(game.door))
        {
            reward -= 50; // Penalty for trying to exit without solving the puzzle
        }

        if (currentStep >= maxSteps)
        {
            reward -= 5; // Minor penalty for each step exceeding the time limit
            game.player.Position = new Vector2(100, game.groundLevel - game.player.Size.Y);// Reset player position
            currentStep = 0;
            IsDone = true; // End the episode if the maximum steps are reached
        }

        if (game.player.Position.X < game.cameraPosition.X || game.player.Position.X > game.cameraPosition.X + game.ScreenWidth ||
            game.player.Position.Y < game.cameraPosition.Y || game.player.Position.Y > game.cameraPosition.Y + game.ScreenHeight)
        {
            reward -= 20; // Penalty for going out of bounds
            game.player.Position = new Vector2(100, game.groundLevel - game.player.Size.Y);// Reset player position
            IsDone = true; // End the episode for out-of-bounds behavior
        }

        if (game.box.Position.X < game.cameraPosition.X || game.box.Position.X > game.cameraPosition.X + game.ScreenWidth ||
            game.box.Position.Y < game.cameraPosition.Y || game.box.Position.Y > game.cameraPosition.Y + game.ScreenHeight)
        {
            reward -= 20; // Penalty for going out of bounds
            game.box.Position = new Vector2(200, game.groundLevel - game.box.Size.Y);// Reset player position
            IsDone = true; // End the episode for out-of-bounds behavior
        }

        if (game.IsIdle())
        {
            reward -= 1; // Penalty for standing still, encouraging active exploration
        }

        // Small positive rewards for exploration
        if (game.IsExploringNewArea())
            reward += 1; // Encourage exploration to discover mechanics or objects

        return (GetState(), reward, IsDone);
    }

    public bool IsDone()
    {
        return game.IsOpen || currentStep >= maxSteps;
    }
}