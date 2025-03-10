using System.Runtime.Serialization.Formatters;
using AI_project_escapeRoom; // Replace 'YourNamespace' with the actual namespace where Game1 is defined
using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Graphics;
using Microsoft.Xna.Framework.Input;
using System.Linq;
using System.Diagnostics;
using System;
using System.IO;
using System.Threading;
using System.Collections.Generic;

class GameEnvironment
{
    private Game1 game;
    private int maxSteps = 5000;
    private int currentStep;
    public List<int> lastPlayerMove;

    public GameEnvironment(Game1 game)
    {
        this.game = game;
        this.currentStep = 0;
        this.lastPlayerMove = new List<int>();
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

    public (double[], double, bool) Step(int action)
    {
        currentStep++;
        double reward = 0;
        bool IsDone = false;

        // Apply action
        switch (action)
        {
            case 0: game.player.Move(new Vector2(-10, 0)); break; // Move left
            case 1: game.player.Move(new Vector2(10, 0)); break;  // Move right
            case 2: if (game.player.IsGrounded) game.player.ApplyForce(new Vector2(0, -250)); game.player.IsGrounded = false; break; // Jump
            case 3: game.player.Grab(game.box); break; // Interact (e.g., pick up the box)
            case 4: game.player.DropHeldBox(); break; // Interact (e.g., drop the box)
        }

        lastPlayerMove.Add(action);
        ///////////////////////////////
        // Rewards for key objectives//
        ///////////////////////////////

        //pick the box
        if (game.player.Intersects(game.box) && game.player.heldBox == null)
        {
            reward += 1; // Reward for picking up the box
            Console.WriteLine("Picked up the box. Reward: +10");
        }

        //placing the box on the button
        if (game.box.Intersects(game.button) && game.player.heldBox == null)
        {
            reward += 10; // Reward for placing the box on the button
            Console.WriteLine("Placed the box on the button. Reward: +10");
        }

        //exiting the room finish goal
        if (game.IsPressed && IsOutOfBounds(game.player))
        {
            reward += 100; // Reward for escaping the room
            IsDone = true;
            Console.WriteLine("Escaped the room. Reward: +100");
        }

        ///////////////////////////////
        // Penalties for incorrect behaviors//
        ///////////////////////////////

        //droping the box for no resone
        if (game.player.heldBox == null && game.previousBoxState == true
        && !game.box.Intersects(game.button))
        {
            reward -= 1; // Penalty for dropping the box for no reason
            Console.WriteLine("Dropped the box for no reason. Penalty: -1");
        }

        //culiding with the walls (not the ground)
        if (!game.player.IsGrounded && game.player.Intersects(game.wall))
        {
            reward -= 1; // Penalty for colliding with the walls
            Console.WriteLine("Collided with the walls. Penalty: -1");
        }

        //repeating actions
        if (RepeatingActions(lastPlayerMove))
        {
            reward -= 1; // Penalty for repeating the same action
            Console.WriteLine("Repeated the same action. Penalty: -1");
        }

        //time penalty
        if (currentStep % 100 == 0)
        {
            reward -= 1; // Penalty for taking too long
            Console.WriteLine("Taking too long. Penalty: -1");
        }

        // Reset if out of bounds
        if (IsOutOfBounds(game.player) || IsOutOfBounds(game.box))
        {
            ResetPlayerAndBox();
            Console.WriteLine("Out of bounds. Resetting player and box.");
        }

        // Maximum steps penalty
        if (currentStep >= maxSteps)
        {
            reward -= 10; // Small penalty for exceeding maximum steps
            ResetPlayerAndBox();
            IsDone = true;
            currentStep = 0;
            Console.WriteLine("Exceeded maximum steps. Penalty: -10");
        }

        Thread.Sleep(1);
        return (GetState(), reward, IsDone);
    }

    public bool RepeatingActions(List<int> actions)
    {
        //find if exist any where in the array a 1000 action copy of the fist elements in array 
        for (int i = 0; i < actions.Count - 1000; i++)
        {
            if (actions.GetRange(i, 1000).SequenceEqual(actions.GetRange(0, 1000)))
            {
                return true;
            }
        }
        return false;
    }
    // Helper methods
    private bool IsOutOfBounds(GameObject obj)
    {
        return obj.Position.X < game.cameraPosition.X || obj.Position.X > game.cameraPosition.X + game.ScreenWidth ||
               obj.Position.Y < game.cameraPosition.Y || obj.Position.Y > game.cameraPosition.Y + game.ScreenHeight;
    }

    private void ResetPlayerAndBox()
    {
        ResetPlayerPosition();
        ResetBoxPosition();
    }

    private void ResetPlayerPosition()
    {
        game.player.Position = new Vector2(100, game.groundLevel - game.player.Size.Y);
    }

    private void ResetBoxPosition()
    {
        game.box.Position = new Vector2(200, game.groundLevel - game.box.Size.Y);
    }

    public bool IsDone()
    {
        return game.IsOpen || currentStep >= maxSteps;
    }
}